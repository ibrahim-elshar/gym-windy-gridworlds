OpenAI gym Windy-Gridworlds
=====================

Implementation of four windy gridworlds environments (Windy Gridworld,
Stochastic Windy Gridworld, Windy Gridworld with King's Moves, Stochastic Windy Gridworld with King's Moves)
from book `Reinforcement Learning: An Introduction
<http://incompleteideas.net/book/the-book-2nd.html>`_
compatible with `OpenAI gym <https://github.com/openai/gym>`_.

Installation
-------------
Install `OpenAI gym <https://github.com/openai/gym>`_.

Clone this repo: 

.. code::

		$ git clone https://github.com/ibrahim-elshar/gym-windy-gridworlds.git
		

Then install this package via

.. code::

		$ pip install -e .



Usage
-----

.. code::

        $ import gym
        $ import gym_windy_gridworlds
        $ env = gym.make('WindyGridWorld-v0')  

``WindyGridWorld-v0``
----------------

Windy Gridworld is as descibed in example 6.5 on page 130, in the book_.
Windy Gridworld is a standard gridworld with start and goal states.
The difference is that there is a crosswind running upward through the 
middle of the grid. Actions are the standard four: up, right, down, and left.
In the middle region the resultant next states are
shifted upward by the "wind" which strength varies from column to column.
The reward is -1 until goal state is reached.

.. _book: http://incompleteideas.net/book/the-book-2nd.html

``StochWindyGridWorld-v0``
---------------------

Stochastic Windy Gridworld is as described above. However,
the effect of the wind "if there is any" is stochastic, sometimes varying
by 1 from the value given for each column.
By default, the probabilities are set uniformly such that a third of the 
time you move one cell according to the wind values as above, but also 
a third of the time you move one cell above that, and another third of the 
time you move one cell below that.

``KingWindyGridworld-v0``
------------

Windy Gridworld with King's moves is the same as Windy Gridworld, however the
agent can move now in 8 possible directions including diagonal moves.

``StochKingWindyGridworld-v0``
------------

Stochastic Windy Gridworld with King's moves is an evironment where the agent can
move in 8 directions including diagonal moves and the wind is stochastic as descibed 
in Stochastic Windy Gridworld.