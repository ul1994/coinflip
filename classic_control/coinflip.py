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
from eth import Series, Eras
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

EPOCH_0 = 0
WORTH_0 = 1000

class CoinFlipEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def load_data(self):
		histfile = '%s/data/USDT_ETH-1800.json' % ethdata_folder
		livefile = '%s/data/public.json' % ethdata_folder

		with open(histfile) as fl:
			histdata = json.load(fl)

		with open(livefile) as fl:
			histdata += json.load(fl)

		series = Series(histdata, sample=5)
		# series.era(Eras.Crash1)
		series.era(Eras.Sine)
		self.series = series

	def init_start_worth(self):
		# worth = 800 + np.random.uniform(low=0, high=200)
		worth = 900 + np.random.uniform(low=0, high=100)
		self.start_worth = worth
		return worth

	def init_start_pos(self):
		# return int(np.random.random_sample() > 0.5)
		return 1

	def __init__(self):
		# Define general world parameters
		# TODO: define exchange fee
		self.load_data()
		self.epoch = EPOCH_0

		position = self.init_start_pos()# initialize with hold state (nothing)
		worth = self.init_start_worth() # enough to purchase at least 1 eth
		self.worth = worth

		# Set some endgame parameters
		self.cap_worth = worth * 50 # can accrue no more than this much
		self.neg_worth = worth / 2.0 # end the game if worth looks bad
		self.cap_position = 1 # hold at most 10 ETH

		# Define action space 3? {0 hold, 1 buy/sell}?
		self.action_space = spaces.Discrete(3)
		self.buy_price = self.series.prices[0] # historical buy price
		self.start_worth -= self.buy_price
		self.exch_fee = 1.0 # exchange fee of at least $1

		# Define obs space - range of possible worldstates
		# 1. all possible positions: 0 eth ... 1 eth
		# REMOVED 2. all possible total worths: 500 ... 50 * 1000
		# 2. price of ETH / 1000 at this time (scaled from 0 ... 1)
		# 3. last buy price

		high = np.array([1, 1, 1])
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
		eth_value = self.series.prices[self.epoch]
		if action == 0: # hold
			reward = 0.0

			self.last_action = action
		elif action == 1 and position < self.cap_position: # buy
			position += 1
			worth -= eth_value
			self.buy_price = eth_value # save last bought price
			reward = 0.0 # some incentive to buying

			self.last_action = action
		elif action == 2 and position > 0: # sell
			print eth_value, self.buy_price
			raw_input(':')
			final_value = eth_value - self.exch_fee
			worth += final_value
			position -= 1
			net_gain = final_value - self.buy_price

			if net_gain > 0:
				reward = 1.0 + net_gain # full reward for gain after sells
				# reward = 1.0 + net_gain ** 2.0 # proportional reward for gain after sells
			else:
				# reward = -1.0 # full punishment of loss in sells
				reward = 0.0 # medium punishment

			self.last_action = action
		else:
			# reward = -0.5 # severely punish invalid moves
			reward = 0.0 # no contribution from invalid moves

			self.last_action = None

		# Changed worldstate wrt. action
		self.worth = worth
		self.state = (position, eth_value / 1000.0, self.buy_price / 1000.0)

		# Check endgame states
		# 1. suffer from a major loss
		# 1.5. in the red
		# 2. hits upper bank limit
		# 3. reaches end of allowed dataset
		done =  self.neg_worth > self.worth \
				or self.worth <= self.start_worth \
				or self.worth > self.cap_worth \
				or self.epoch == len(self.series.prices)
		done = bool(done)
		self.epoch += 1 # incrememt world time

		return np.array(self.state), reward, done, {}

	def _reset(self):
		# FIXME: Correctly reinitialize a random start state
		self.epoch = EPOCH_0
		worth = self.init_start_worth()
		self.worth = worth
		position = self.init_start_pos()
		self.buy_price = self.series.prices[0]
		self.state = [position, self.series.prices[0] / 1000.0, self.buy_price / 1000.0]

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
			self.gui_position = rendering.FilledPolygon([(0,0), (0,20), (20,20), (20,0)])
			self.gui_position.set_color(.9,.0,.0)
			self.positiontrans = rendering.Transform()
			self.gui_position.add_attr(self.positiontrans)
			self.viewer.add_geom(self.gui_position)

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

		for ii in range(min(self.epoch, len(self.tickers))):
			time_i = self.epoch - 1 # render happens after first step
			maxprice = 600.0
			tind = time_i
			if self.epoch >= len(self.tickers):
				tind = ii
				time_i = self.epoch - len(self.tickers) + ii
			ticker, trans = self.tickers[tind]

			pscale = self.series.prices[time_i] / float(maxprice)
			pixheight = float(pscale * float(screen_height))
			trans.set_translation(0, pixheight)
			ticker.set_color(.8, .8, .8)

		last_ticker, last_trans = self.tickers[ii]
		if self.last_action == 1:
			last_ticker.set_color(.3, .9, .3)
		elif self.last_action == 2:
			last_ticker.set_color(.9, .3, .3)

		for ii in range(min(self.epoch, len(self.tickers)) + 1, len(self.tickers)):
			self.tickers[ii][0].set_color(.3, .3, .3)


		position, _, _ = self.state
		pixworth = int((self.worth - self.start_worth) / 10.0)# worth
		self.statustrans.set_translation(0, pixworth)
		self.positiontrans.set_translation(0, 0 if position == 0 else 100)

		if self.worth > self.past_max_returns:
			self.past_max_returns = self.worth
			self.maxline_trans.set_translation(0, pixworth)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
