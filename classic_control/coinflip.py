"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

# ULZ

ethdata_folder = '/Users/ulzee/nyu/eth/external/eth-history'
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
		series.era(Eras.Crash1)
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

		# Define action space 3? {0 hold, 1 buy/sell}?
		self.action_space = spaces.Discrete(3)
		self.buy_price = 0.0 # historical buy price

		# Define obs space - range of possible worldstates
		# 1. all possible positions: 0 eth ... 1 eth
		# REMOVED 2. all possible total worths: 500 ... 50 * 1000
		# 3. possible range of price of ETH / 1000 (scaled from 0 ... 1)

		high = np.array([1, 1])
		low = np.array([0, 0])
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
		self.epoch += 1 # incrememt world time
		state = self.state

		# Correctly unpack new worldstate
		position, _ = state
		worth = self.worth

		# Correclty adjust world
		eth_value = self.series.prices[self.epoch]
		if action == 0: # hold
			# Holding has 0 reward because it is a neutral action
			#  that doesn't contribute to the optimization
			reward = 0.0
		elif action == 1 and position == 0: # buy
			# buy
			# FIXME: what is a consistent reward for a buy?
			worth -= eth_value
			self.buy_price = eth_value
			position = 1 # +=
			reward = 1.0
		elif action == 2 and position == 1: # sell
				# sell
				worth += eth_value
				position = 0
				# FIXME: define appropriate reward for selling
				net_gain = eth_value - self.buy_price
				reward = 1.0
				if net_gain > 0:
					reward = 1.0
				else:
					reward = -1.0 # punish loss in sells
				# reward = net_gain # FIXME: normalize this?
		else:
			reward = -0.5 # severely punish invalid moves

		# Changed worldstate wrt. action
		self.worth = worth
		self.state = (position, eth_value / 1000.0)

		# Check endgame thresholds
		done =  self.neg_worth > self.worth \
				or self.start_worth > self.worth \
				or self.worth > self.cap_worth \
				or self.epoch == len(self.series.prices)
		done = bool(done)

		return np.array(self.state), reward, done, {}

	def _reset(self):
		# FIXME: Correctly reinitialize a random start state
		self.epoch = EPOCH_0
		worth = self.init_start_worth()
		position = self.init_start_pos()
		self.state = [position, self.series.prices[0] / 1000.0]

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
			self.baseline = rendering.FilledPolygon([(0,0), (0,2), (screen_width - 1,2), (screen_width - 1,0)])
			self.baseline.set_color(.3,.8,.5)
			self.baselinetrans = rendering.Transform()
			self.baseline.add_attr(self.baselinetrans)
			self.viewer.add_geom(self.baseline)

			# add position status indicator
			self.gui_position = rendering.FilledPolygon([(0,0), (0,20), (20,20), (20,0)])
			self.gui_position.set_color(.9,.0,.0)
			self.positiontrans = rendering.Transform()
			self.gui_position.add_attr(self.positiontrans)
			self.viewer.add_geom(self.gui_position)

			# add a continuous ticker graph
			self.tickers = []
			for ii in range(screen_width / bar_w):
				wt, ht = bar_w, 2
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
			time_i = self.epoch
			maxprice = 600.0
			tind = time_i
			if self.epoch >= len(self.tickers):
				tind = ii
				time_i = self.epoch - len(self.tickers) + ii
			ticker, trans = self.tickers[tind]

			pscale = self.series.prices[time_i] / float(maxprice)
			pixheight = float(pscale * float(screen_height))
			trans.set_translation(0, pixheight)
			ticker.set_color(.8, .6, .3)

		for ii in range(min(self.epoch, len(self.tickers)) + 1, len(self.tickers)):
			self.tickers[ii][0].set_color(.3, .3, .3)

		position, _ = self.state
		pixworth = int(self.worth / 10.0) # worth
		self.statustrans.set_translation(0, pixworth)
		self.positiontrans.set_translation(0, 0 if position == 0 else 100)
		self.baselinetrans.set_translation(0, int(self.start_worth / 10.0))

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
