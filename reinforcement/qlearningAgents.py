# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)
		"*** YOUR CODE HERE ***"
		self.qValues = util.Counter()
	def getQValue(self, state, action):
		"*** YOUR CODE HERE ***"
		return self.qValues[(state, action)]
		util.raiseNotDefined()

	def getValue(self, state):
		"*** YOUR CODE HERE ***"
		possible_QValues = util.Counter()
		for action in self.getLegalActions(state):
			possible_QValues[action] = self.getQValue(state, action)
		return possible_QValues[possible_QValues.argMax()]
		util.raiseNotDefined()

	def getPolicy(self, state):
		"*** YOUR CODE HERE ***"
		possible_QValues = util.Counter()
		possibleActions = self.getLegalActions(state)
		if len(possibleActions) == 0:
			return None

		for action in possibleActions:
			possible_QValues[action] = self.getQValue(state, action)

		if possible_QValues.totalCount() == 0:
			return random.choice(possibleActions)
		else:
			return possible_QValues.argMax()
		util.raiseNotDefined()


	def getAction(self, state):
		# Pick Action
		legalActions = self.getLegalActions(state)
		action = None
		"*** YOUR CODE HERE ***"
		if len(legalActions) > 0:
			if util.flipCoin(self.epsilon):
				action = random.choice(legalActions)
			else:
				action = self.getPolicy(state)
		return action
		util.raiseNotDefined()


	def update(self, state, action, nextState, reward):
		"*** YOUR CODE HERE ***"
		self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
		util.raiseNotDefined()

class PacmanQAgent(QLearningAgent):
	"Exactly the same as QLearningAgent, but with different default parameters"

	def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		self.index = 0  # This is always Pacman
		QLearningAgent.__init__(self, **args)

	def getAction(self, state):
		action = QLearningAgent.getAction(self,state)
		self.doAction(state,action)
		return action


class ApproximateQAgent(PacmanQAgent):
	def __init__(self, extractor='IdentityExtractor', **args):
		self.featExtractor = util.lookup(extractor, globals())()
		PacmanQAgent.__init__(self, **args)
		# You might want to initialize weights here.
		"*** YOUR CODE HERE ***"
		self.weights = util.Counter()


	def getQValue(self, state, action):
		"*** YOUR CODE HERE ***"
		qValue = 0.0
		features = self.featExtractor.getFeatures(state, action)
		for key in features.keys():
			qValue += (self.weights[key] * features[key])
		return qValue
		util.raiseNotDefined()

	def update(self, state, action, nextState, reward):
		"*** YOUR CODE HERE ***"
		features = self.featExtractor.getFeatures(state, action)
		possibleStateQValues = []
		for act in self.getLegalActions(state):
			possibleStateQValues.append(self.getQValue(state, act))
		for key in features.keys():
			self.weights[key] += self.alpha * (reward + self.discount * ((1-self.epsilon)*self.getValue(nextState)+(self.epsilon/len(possibleStateQValues))*(sum(possibleStateQValues))) - self.getQValue(state, action)) * features[key]
		util.raiseNotDefined()

	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		PacmanQAgent.final(self, state)
		# did we finish training?
		if self.episodesSoFar == self.numTraining:
			# you might want to print your weights here for debugging
			"*** YOUR CODE HERE ***"
			pass
