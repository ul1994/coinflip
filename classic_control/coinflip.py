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

		series = Series(histdata, sample=10)
		series.era(Eras.Modern)
		self.series = series

	def __init__(self):
		# Define general world parameters
		# TODO: define exchange fee
		self.load_data()
		self.epoch = EPOCH_0

		self.position = int(np.random.random_sample() > 0.5) # initialize with hold state (nothing)
		self.worth = WORTH_0 # enough to purchase at least 1 eth
		self.whist = [WORTH_0]

		# Set some baseline endgame parameters
		# Angle at which to fail the episode
		self.toplim = self.worth * 50
		self.neg_worth = self.worth / 2.0

		# Define action space 3? {0 sell, 1 hold, 2 buy}?
		self.action_space = spaces.Discrete(3)

		# Define obs space - range of possible worldstates
		# 1. all possible positions: 0 eth ... 1 eth
		# 2. all possible total worths: 500 ... 50 * 1000
		high = np.array([1, self.toplim])
		low = np.array([0, self.neg_worth])
		self.observation_space = spaces.Box(low, high)

		self._seed()
		self.viewer = None
		self.state = None

		self.steps_beyond_done = None

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		self.epoch += 1 # incrememt world time
		state = self.state

		# Correctly unpack new worldstate
		position, worth = state

		# Correclty adjust world
		self.buy_price = 0.0
		eth_value = self.series.prices[self.epoch]
		if self.position == 1 and action == 0:
			# sell
			worth += eth_value
			self.position = 0
			# TODO: define appropriate reward for selling
			net_gain = eth_value - self.buy_price
			reward = net_gain # TODO: normalize this?
		elif self.position == 1 and action == 1:
			# hold
			# baseline worth does not change
			# self.position # position does not change
			# TODO: define appropriate reward for holding
			reward = 0.0
		elif self.position == 0 and action == 2:
			# buy
			worth -= eth_value
			self.buy_price = eth_value
			self.position += 1
			reward = 1.0
		else:
			# FIXME: reward is same as holding. How do we prevent spamming invalid moves.
			# Invalid move ... no reward? neg reward?
			# TODO: define appropriate reward for invalid
			# Soft punish invalid moves?
			reward = -0.1

		# Changed worldstate wrt. action
		self.state = (position, worth)

		# Check endgame thresholds
		done =  self.neg_worth > worth \
				or self.worth > self.toplim \
				or self.epoch == len(self.series.prices)
		done = bool(done)

		return np.array(self.state), reward, done, {}

	def _reset(self):
		# FIXME: Correctly reinitialize a random start state
		self.epoch = EPOCH_0
		self.whist = [WORTH_0]
		self.position = int(np.random.random_sample() > 0.5)
		self.state = [self.position, WORTH_0]
		# self.position = int(np.random.random_sample() > 0.5)
		self.steps_beyond_done = None
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

			# add a continuous ticker graph
			self.tickers = []
			for ii in range(screen_width / bar_w):
				wt, ht = bar_w, 1
				lhs = bar_w * ii
				rhs = lhs + wt
				ticker = rendering.FilledPolygon([(lhs,0), (lhs,ht), (rhs,ht), (rhs,0)])
				ticker.set_color(.3,.3,.3)
				tickertrans = rendering.Transform()
				ticker.add_attr(tickertrans)
				self.viewer.add_geom(ticker)
				self.tickers.append((ticker, [tickertrans, 1]))

		if self.state is None: return None

		for ii in range(min(self.epoch, len(self.tickers))):
			time_i = self.epoch
			maxprice = 600.0
			tind = time_i
			if self.epoch >= len(self.tickers):
				tind = ii
				time_i = self.epoch - len(self.tickers) + ii
			ticker, args = self.tickers[tind]
			trans, ht = args

			# Resize to ht of 1
			trans.set_scale(1.0, 1.0 / float(ht))
			pscale = self.series.prices[time_i] / float(maxprice)
			pixheight = float(pscale * float(screen_height))
			trans.set_scale(1.0, pixheight)

			ticker.set_color(.8, .6, .3)
			self.tickers[tind][1][1] = pixheight

		pixworth = int(self.whist[-1] / 10.0)
		self.statustrans.set_translation(0, pixworth)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
