# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import random

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
	self.mdp = mdp
	self.discount = discount
	self.iterations = iterations
	self.values = util.Counter() # A Counter is a dict with default 0
	print("a")
	for itera in range(self.iterations):
		for states in self.mdp.getStates():
			Actions=self.mdp.getPossibleActions(states)
			values_actions=util.Counter()
			for acti in Actions:
				transitions=self.mdp.getTransitionStatesAndProbs(states, acti)
				value_state=0
				for trans in transitions:
					value_state=value_state+(trans[1] * (self.mdp.getReward(states, acti, trans[0]) + self.discount * self.values[trans[0]]))
				values_actions[acti]=value_state
		self.values[states] = values_actions[values_actions.argMax()]

    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
	transitions = self.mdp.getTransitionStatesAndProbs(state, action)
	Q_Value = 0
	for transition in transitions:
		Q_Value += transition[1] * self.mdp.getReward(state, action, transition[0]) + self.values[transition[0]]
	return Q_Value
	util.raiseNotDefined()

  def getPolicy(self, state):
	if self.mdp.isTerminal(state):
		return None
	possible_actions = self.mdp.getPossibleActions(state)
	values_actions = util.Counter()
	for action in possible_actions:
		transitions = self.mdp.getTransitionStatesAndProbs(state, action)
		value_state = 0
		for transition in transitions:
			value_state += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
		values_actions[action] = value_state
	
	if values_actions.totalCount() == 0:
		return possible_actions[int(random.random() * len(possible_actions))]
	else:
		value = values_actions.argMax()
		return value
	util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
