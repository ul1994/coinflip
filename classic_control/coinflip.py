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
		# TODO: Define general world parameters
		self.load_data()
		self.epoch = 0

		self.state = 0
		self.worth = 1000 # enough to purchase at least 1 eth

		# Set some baseline endgame parameters
		# Angle at which to fail the episode
		self.toplim = 1000 * 50
		self.neg_worth = self.worth / 2.0

		# Define action space 3? {hold, buy, sell}?
		self.action_space = spaces.Discrete(3)

		# Define obs space - bounds for the results of taking actions
		# example for cartpole: (min angle, max angle) ... 1-dimensional
		# observation is the accumulated value
		high = np.array([self.toplim])
		low = np.array([self.neg_worth])
		self.observation_space = spaces.Box(low, high)

		self._seed()
		self.viewer = None
		self.state = None

		self.steps_beyond_done = None

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		# Simulate the reaction to some action
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		# TODO: Correctly unpack new worldstate
		self.epoch += 1
		x, x_dot, theta, theta_dot = state
		# TODO: Correclty adjust world
		force = self.force_mag if action==1 else -self.force_mag
		costheta = math.cos(theta)
		sintheta = math.sin(theta)
		temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
		xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
		x  = x + self.tau * x_dot
		x_dot = x_dot + self.tau * xacc
		theta = theta + self.tau * theta_dot
		theta_dot = theta_dot + self.tau * thetaacc

		# TODO: Correctly change worldstate wrt. action
		self.state = (x,x_dot,theta,theta_dot)
		# TODO: Verify if minimal threshold is hit
		done =  x < -self.x_threshold \
				or x > self.x_threshold \
				or theta < -self.theta_threshold_radians \
				or theta > self.theta_threshold_radians \
				or self.epoch == len(self.series.prices)
		done = bool(done)

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0

		return np.array(self.state), reward, done, {}

	def _reset(self):
		# TODO: Correctly reinitialize a random start state
		self.epoch = 0
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
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

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
