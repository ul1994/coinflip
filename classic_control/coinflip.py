"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

# ULZ
import os
ethdata_folder = os.environ['ETH_HISTORY']
import sys, json
sys.path.append(ethdata_folder)
from eth import Segments, Series, Eras
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from rl.core import TRAINING_MODE

logger = logging.getLogger(__name__)

EPOCH_0 = 0
# SEGMENT_TYPE = '15'
SEGMENT_TYPE = '60'
SEGMENT_MODE = 'Full'
# SEGMENT_MODE = 'Disc'
SEGMENT_SIZE = 24 * 16
f = 1.0
GAIN_DAMPNER = 100.0
# GAIN_DAMPNER = 10.0
EXCH_FEE = 0.0025
# LOSS_TOLERANCE = 0.95
LOSS_TOLERANCE = 0.98
# LOSS_TOLERANCE = 0.99
# LOSS_TOLERANCE = 1.0
# ABS_LOSS_TOL = 25.0
# ABS_LOSS_TOL = 0.0
ABS_LOSS_TOL = 30.0
# LOSS_TOLERANCE = 1.0
DEBUG_MODE = False
KEEP_SEG = False
REWARD_MODE = 'max'
# REWARD_MODE = 'sell'

class CoinFlipEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def load_segments(self):
		datapath = ethdata_folder
		handle = Segments(datapath)
		return handle

	def load_series(self):
		histfile = '%s/data/hist.json' % ethdata_folder

		with open(histfile) as fl:
			histdata = json.load(fl)

		series = Series(histdata, sample=5)
		# series.era(Eras.Crash1)
		# series.era(Eras.Sine)
		series.era(Eras.NoisySine)
		self.series = series

	def init_start_worth(self):
		# worth = 800 + np.random.uniform(low=0, high=200)
		worth = 900  #+ np.random.uniform(low=0, high=100)
		# self.start_worth = worth
		return worth

	def init_start_pos(self):
		# return int(np.random.random_sample() > 0.5)
		return 1

	def __init__(self):
		self.segs = self.load_segments()
		self.load_series()
		# tlen, vlen = self.segs.get_size('060')
		# self.train_len = tlen
		# self.val_len = vlen

		# Define action space 3? {0 hold, 1 buy/sell}?
		self.action_space = spaces.Discrete(3)

		# Define obs space - range of possible worldstates
		# 1. all possible positions: 0 eth ... 1 eth
		# REMOVED 2. all possible total worths: 500 ... 50 * 1000
		# 2. price of ETH / 1000 at this time (scaled from 0 ... 1)
		# 3. ratio of (last buy price) / (current price)
		high = np.array([1, 1, 10.0])
		low = np.array([0, 0, 0])
		self.observation_space = spaces.Box(low, high)

		self._seed()
		self.reset()
		self.viewer = None
		self.state = None

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state

		# Correctly unpack new worldstate
		position, _, _ = state
		worth = self.worth

		# Correclty adjust world
		if self.last_buy != None: self.last_buy += 1
		self.bad_deal = False
		eth_value = self.segment[self.epoch]
		if action == 0: # hold
			# TODO: Some reward if eth went up?
			reward = 0.0
			self.last_action = action
		elif action == 1 and position < self.cap_position: # buy
			position += 1
			worth -= eth_value
			self.buy_price = eth_value # save last bought price
			reward = 0.0 # no direct incentive, but critical to sell action
			self.last_action = action
			self.last_buy = 0 # start tracking last buy
		elif action == 2 and position > 0: # sell
			# print worth, self.max_worth, eth_value
			final_value = eth_value * (1.0 - EXCH_FEE)
			worth += final_value
			position -= 1

			if REWARD_MODE == 'max':
				gain = worth - self.max_worth
				if self.max_worth < worth:
					self.max_worth = worth
				gain_reward = (gain / GAIN_DAMPNER) ** 2.0
			elif REWARD_MODE == 'sell':
				gain = final_value - self.buy_price
				gain_reward = (gain / GAIN_DAMPNER) ** 2.0

			if gain > 0:
				# reward = 1.0 + gain_reward # full reward for gain after sells
				# plus additional reward proportional to net gain
				reward = gain_reward
			else:
				# loss_reward = (net_gain / GAIN_DAMPNER) ** 2.0
				reward = -gain_reward
				reward = 0
				if gain < -ABS_LOSS_TOL:
					self.bad_deal = True

			self.last_action = action
			self.last_buy = None # stop tracking last buy
			self.txn += 1
		else:
			reward = 0.0 # no contribution from invalid moves
			self.last_action = None

		# Changed worldstate wrt. action
		self.worth = worth
		self.worth_hist.append(self.worth - self.start_worth)
		if self.worth > self.wmax: self.wmax = self.worth
		if self.worth < self.wmin: self.wmin = self.worth
		self.state = (position, eth_value / 1000.0, self.buy_price / eth_value)
		self.epoch += 1 # incrememt world time

		# Check endgame states
		# 1. A sell action causes you to go into red
		# 2. A lot of steps since last buy and you are in red
		# 3. end of train data
		justSold = self.last_action == 2
		heldForLong = self.last_buy > 20 # extended period of holding
		# inRed = self.worth < self.start_worth * LOSS_TOLERANCE
		# inRed = self.worth < self.max_worth * LOSS_TOLERANCE
		inRed = self.worth < self.max_worth - ABS_LOSS_TOL
		# print justSold and inRed, self.bad_deal,heldForLong and inRed, self.epoch == len(self.segment)
		done =  justSold and inRed \
				or heldForLong and inRed \
				or self.epoch == len(self.segment) \
				or self.bad_deal
		done = bool(done)
		info = {}
		info['net'] = self.worth - self.start_worth
		info['min'] = self.wmin - self.start_worth
		info['max'] = self.wmax - self.start_worth
		info['txn'] = self.txn
		info['series'] = json.dumps(self.segment[:self.epoch])
		info['worth'] = json.dumps(self.worth_hist)
		self.action_hist.append(self.last_action)
		info['actions'] = json.dumps(self.action_hist)
		info['epoch'] = self.epoch
		return np.array(self.state), reward, done, info

	def _reset(self):
		with open(TRAINING_MODE) as fl:
			mode = fl.readline().strip()
		# FIXME: Correctly reinitialize a random start state
		self.epoch = EPOCH_0
		if DEBUG_MODE:
			if mode == 'val':
				self.series.era(Eras.Sine)
				self.segment = self.series.prices
			elif mode == 'train':
				self.segment = self.series.prices
				self.series.era(Eras.NoisySine)
			else:
				raise Exception('Unknown TRAINING_MODE!')
		else:
			# use_seg = self.segs.train.p60 if SEGMENT_TYPE == '60' else self.segs.train.p15
			# self.segment = use_seg
			if mode == 'val':
				self.segment = self.segs.val.p60

				# self.segment = self.segs.get_long(use_seg, size=200, repeat=1) # 96
			# elif mode == 'train_test':
			# 	self.segs.long_orders = []
			# 	self.segs.repeat = 0
			# 	self.segment = self.segs.get_long(self.segs.train.p60, size=200, repeat=1) # 96
			else:
				if SEGMENT_MODE == 'Disc':
					self.segment = self.segs.get_long(self.segs.train.p60, size=200, repeat=50) # 96
				elif SEGMENT_MODE == 'Full':
					self.segment = self.segs.train.p60

		self.action_hist = []
		self.worth_hist = []
		position = self.init_start_pos() # initialize with hold state (nothing)
		worth = self.init_start_worth() # enough to purchase at least 1 eth
		self.cap_worth = worth * 50 # can accrue no more than this much
		self.cap_position = 1 # hold at most 10 ETH
		self.buy_price = self.segment[self.epoch] # historical buy price
		self.max_worth = worth # before buy price
		self.start_worth = worth
		worth -= self.buy_price
		self.fail_loss = 300
		self.last_buy = None
		self.worth = worth
		self.txn = 0

		self.wmin = 100000000000000
		self.wmax = -100000000000000

		# Set some endgame parameters
		# starting ETH position
		# first price
		# starting buy-sell ratio (is 1)
		self.state = [position, 0, 1.0]

		return np.array(self.state)

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 600
		screen_height = 400

		bar_w = 2
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)

			# add worth status indicator
			self.status = rendering.FilledPolygon([(0,0), (0,2), (screen_width - 1,2), (screen_width - 1,0)])
			self.status.set_color(.8,.3,.8)
			self.statustrans = rendering.Transform()
			self.status.add_attr(self.statustrans)
			self.viewer.add_geom(self.status)

			# add initial worth line
			self.past_max_returns = -1
			maxline = rendering.FilledPolygon([(0,0), (0,2), (screen_width - 1,2), (screen_width - 1,0)])
			maxline.set_color(.3,.8,.5)
			maxlinetrans = rendering.Transform()
			maxline.add_attr(maxlinetrans)
			maxlinetrans.set_translation(0, 0)
			self.maxline_trans = maxlinetrans
			self.viewer.add_geom(maxline)

			# add price lines
			for ii in range(1, 3):
				pline = rendering.FilledPolygon([(0,0), (0,2), (screen_width - 1,2), (screen_width - 1,0)])
				pline.set_color(.8,.8,.3)
				plinetrans = rendering.Transform()
				pline.add_attr(plinetrans)
				plinetrans.set_translation(0, 100 * ii)
				self.viewer.add_geom(pline)

			# add position status indicator
			# self.gui_position = rendering.FilledPolygon([(0,0), (0,20), (20,20), (20,0)])
			# self.gui_position.set_color(.9,.0,.0)
			# self.positiontrans = rendering.Transform()
			# self.gui_position.add_attr(self.positiontrans)
			# self.viewer.add_geom(self.gui_position)

			# add a continuous ticker graph
			self.tickers = []
			for ii in range(screen_width / bar_w):
			# for ii in range(3):
				wt, ht = bar_w, 4
				lhs = bar_w * ii
				rhs = lhs + wt
				ticker = rendering.FilledPolygon([(lhs,0), (lhs,ht), (rhs,ht), (rhs,0)])
				ticker.set_color(.3,.3,.3)
				tickertrans = rendering.Transform()
				ticker.add_attr(tickertrans)
				self.viewer.add_geom(ticker)
				self.tickers.append((ticker, tickertrans))

		if self.state is None: return None

		maxprice = np.max(self.segment)
		minprice = np.min(self.segment)
		pricerange = maxprice - minprice
		for ii in range(min(self.epoch, len(self.tickers))):
			time_i = self.epoch - 1 # render happens after first step
			tind = time_i
			if self.epoch >= len(self.tickers):
				tind = ii
				time_i = self.epoch - len(self.tickers) + ii
			ticker, trans = self.tickers[tind]

			pscale = (self.segment[time_i] - minprice) / float(pricerange)
			pixheight = float(pscale * float(screen_height - 50.0))
			trans.set_translation(0, pixheight)
			ticker.set_color(.8, .8, .8)

		last_ticker, last_trans = self.tickers[ii]
		if self.last_action == 1:
			last_ticker.set_color(.3, .9, .3)
		elif self.last_action == 2:
			last_ticker.set_color(.9, .3, .3)

		for ii in range(min(self.epoch, len(self.tickers)) + 1, len(self.tickers)):
			self.tickers[ii][0].set_color(.3, .3, .3)

		# position, _, _ = self.state
		pixworth = int((self.worth - self.start_worth) / 1.0)# worth
		self.statustrans.set_translation(0, pixworth)
		# self.positiontrans.set_translation(0, 0 if position == 0 else 100)

		if self.worth > self.past_max_returns:
			self.past_max_returns = self.worth
			self.maxline_trans.set_translation(0, pixworth)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
