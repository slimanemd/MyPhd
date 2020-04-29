#https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
#https://github.com/wagonhelm/Reinforcement-Learning-Introduction/blob/master/Reinforcement%20Learning%20Introduction.ipynb
#https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
#https://github.com/2017-fall-DL-training-program/Setup_tutorial/blob/master/OpenAI-gym-install.md
#https://qiita.com/goodboy_max/items/f11bc4bd71e0e2e1cd37
#https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
#https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-for-python-3
#https://www.tecmint.com/install-pip-in-linux/
#https://medium.freecodecamp.org/how-to-build-an-ai-game-bot-using-openai-gym-and-universe-f2eb9bfbb40a
#https://github.com/EN10
#http://www.pinchofintelligence.com/getting-started-openai-gym/
#https://websiteforstudents.com/install-proprietary-nvidia-gpu-drivers-on-ubuntu-16-04-17-10-18-04/
#
#https://github.com/jaimeps/docker-rl-gym
#https://medium.com/coinmonks/preparing-a-headless-environment-for-openais-gym-with-docker-and-tensorflow-1bd0e0d31663
#https://medium.com/curiouscaloo/a-gpu-ready-docker-container-for-openai-gym-development-with-tensorflow-9be3d61504cb
#https://github.com/wagonhelm/Reinforcement-Learning-Introduction
#https://github.com/duckietown/gym-duckietown   Self-driving car simulator for the Duckietown universe
#https://github.com/DrSnowbird/docker-openai-gym


# Introduction to reinforcement learning and OpenAI Gym 
# A demonstration of basic reinforcement learning problems.
# By Justin Francis July 13, 2017

# Those interested in the world of machine learning are aware of the capabilities of reinforcement-learning-based AI. The past few years have seen many breakthroughs using reinforcement learning (RL). The company DeepMind combined deep learning with reinforcement learning to achieve above-human results on a multitude of Atari games and, in March 2016, defeated Go champion Le Sedol four games to one. Though RL is currently excelling in many game environments, it is a novel way to solve problems that require optimal decisions and efficiency, and will surely play a part in machine intelligence to come.

#OpenAI was founded in late 2015 as a non-profit with a mission to “build safe artificial general intelligence (AGI) and ensure AGI's benefits are as widely and evenly distributed as possible.” In addition to exploring many issues regarding AGI, one major contribution that OpenAI made to the machine learning world was developing both the Gym and Universe software platforms.

#Gym is a collection of environments/problems designed for testing and developing reinforcement learning algorithms—it saves the user from having to create complicated environments. Gym is written in Python, and there are multiple environments such as robot simulations or Atari games. There is also an online leaderboard for people to compare results and code.

#Reinforcement learning, explained simply, is a computational approach where an agent interacts with an environment by taking actions in which it tries to maximize an accumulated reward. Here is a simple graph, which I will be referring to often:

#An agent in a current state (St) takes an action (At) to which the environment reacts and responds, returning a new state(St+1) and reward (Rt+1) to the agent. Given the updated state and reward, the agent chooses the next action, and the loop repeats until an environment is solved or terminated.

#OpenAI’s Gym is based upon these fundamentals, so let’s install Gym and see how it relates to this loop. We’ll get started by installing Gym using Python and the Ubuntu terminal.

# sudo apt-get install -y python3-numpy python3-dev python3-pip cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
# cd ~
# git clone https://github.com/openai/gym.git
# cd gym
# sudo pip3 install -e '.[all]'


import numpy as np
import gym			#Next, open Python3 in terminal and import Gym.
import time

#
def do_scenario00():
    env = gym.make("Taxi-v2")    	#1st, need ENV. load ENV*[taxi].
    env.reset()                    	#To initialize ENV, we must reset it.

    # ou will notice that resetting ENV will return an integer. This number will be our initial ST. 
    # All possible states in this ENV are represented by an integer ranging from 0 to 499. 

    env.observation_space.n		# get the total number of possible ST*
    env.render()    		# to visualize ST_cur

    # In this ENV 
    # - the yellow square represents the taxi, 
    # - the (“|”) represents a wall, 
    # - the blue letter represents the pick-up location, and 
    # - the purple letter is the drop-off location. 
    # The taxi will turn green when it has a passenger aboard. 
    # While we see colors and shapes that represent the environment, 
    # the algorithm does not think like us and only understands a flattened state, in this case an integer.

    env.action_space.n 		# ACT* available to AGT

    # This shows us there are a total of six actions available. 
    # The 6 possible ACT* : down (0), up (1), right (2), left (3), pick-up (4), and drop-off (5).

    env.env.s = 114 		# override the current state to 114.
    env.render()

    env.step(1)			# move up.
    env.render()   			# return four variables : (14, -1, False, {'prob': 1.0})
    return env


#These four variables are: 
# - the new state (St+1 = 14), 
# - reward (Rt+1 = -1), 
# - a boolean stating whether the environment is terminated or done, and 
# - extra info for debugging. 
# Every Gym ENV will return these same 4VAR* after _ACT is taken, as they are the core VAR* of RL_PB_.

# What do you expect ENV would return if you were to move left? It would, of course, give the exact same return as before. ENV always gives a -1 RWD for each STP in order for AGT to try and find the quickest SOL possible. If you were measuring your total accumulated RWD, constantly running into a wall would heavily penalize your final RWD. ENV will also give a -10 RWD every time you incorrectly pick up or drop off a passenger.

# So, how can we solve ENV?

# One surprising way you could solve this ENV is to choose randomly among the six possible ACT*. ENV is considered solved when you successfully pick up a passenger and drop them off at their desired location. Upon doing this, you will receive _RWD of 20 and done will equal True. 
# The odds are small, but it’s still possible, and given enough random ACT* you will eventually luck out. A core part of evaluating any AGT’s performance is to compare it to a completely random AGT. In a Gym ENV, you can choose a random ACT using env.action_space.sample(). You can create a loop that will do random ACT* until ENV is solved. We will put a counter in there to see how many steps it takes to solve the ENV.

#
def do_scenario01(env0, delay=1):
    env = gym.make(env0)    	#1st, need ENV. load ENV*[taxi].
    env.reset()                    	#To initialize ENV, we must reset it.
    
    state = env.reset()
    counter = 0  ;reward = None ;done=False  
    while not done:                          # counter < 20: reward != 20
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()   ; time.sleep(delay)
        counter += 1
        print("(step, s, a, r, d) = (", counter, ",", state, ",", action,",", reward,",", done ,")" )
    #print(counter)
    env.env.close()

# 2. To check all env available, uninstalled ones are also shown
#from gym import envs 
#print(envs.registry.all())

# On average, a completely random POL will solve this ENV in about 2000+ STP*, so in order to maximize our RWD, we will have to have the ALGO remember its ACT* & their associated RWD*. In this case, ALGO’s MEM is going to be a Q ACT VAL TAB.
# To manage this Q TAB, we will use a NumPy array. The size of this TAB will be the number of STA* (500) by the number of possible ACT* (6).

#Over multiple episodes of trying to solve the problem, we will be updating our Q values, slowly improving our algorithm’s efficiency and performance. We will also want to track our total accumulated reward for each episode, which we will define as G.

# #    
def playOneEpisode(env, Q, totalReward, alpha, delay =0.1):
    done = False; totalReward=0; reward = 0 ;state = env.reset()
    while done != True:
        action = np.argmax(Q[state]) 							#1
        stateNext, reward, done, info = env.step(action) 				#2
        Q[state,action] += alpha * (reward + np.max(Q[stateNext]) - Q[state,action]) 	#3
        totalReward += reward
        env.render() ; time.sleep(delay)
        print("(step, s, a, r, d) = ( ", state, ", ", action,", ", stateNext, ", ", reward,", ", Q[state,action], ", ", totalReward,", ", done ,")" )  
        state = stateNext 
        

#
def do_scenario03(pEnvName, pNbEpisodes=1, pAlpha = 0.618):
    env = gym.make(pEnvName)   	#1st, need ENV. load ENV*[taxi].    env.reset()                    	#To initialize ENV, we must reset it.
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    totalReward = 0; pAlpha = 0.618	#Learning rate, can implement a basic Q learning algorithm.
    for episode in range(0,pNbEpisodes):
        playOneEpisode(env, Q, totalReward, pAlpha)
        if episode % 50 == 0:
            print('Episode {} Total Reward: {}'.format(episode,totalReward))


#do_scenario00()
#do_scenario01("Taxi-v2")
#do_scenario01("Acrobot-v1",0.1)
#do_scenario01("SpaceInvaders-v0",0.1)
#do_scenario03("Taxi-v2")
do_scenario01("MsPacman-v0",0.01)


# This code alone will solve the environment. There is a lot going on in this code, so I will try and break it down.
#First (#1): The agent starts by choosing an action with the highest Q value for the current state using argmax. 
#Argmax will return the index/action with the highest value for that state. Initially, our Q table will be all zeros. But, after every step, the Q values for state-action pairs will be updated.

#Second (#2): The agent then takes action and we store the future state as state2 (St+1). This will allow the agent to compare the previous state to the new state.

#Third (#3): We update the state-action pair (St , At) for Q using the reward, and the max Q value for state2 (St+1). This update is done using the action value formula (based upon the Bellman equation) and allows state-action pairs to be updated in a recursive fashion (based on future values). See Figure 2 for the value iteration update.

#Q-Learning Formula
#Figure 2. Q-Learning Formula. Source: By Gregz448, CC0, on Wikimedia Commons.
#Following this update, we update our total reward G and update state (St) to be the previous state2 (St+1) so the loop can begin again and the next action can be decided.

#After so many episodes, the algorithm will converge and determine the optimal action for every state using the Q table, ensuring the highest possible reward. We now consider the environment problem solved.

#Now that we solved a very simple environment, let’s move on to the more complicated Atari environment—Ms. Pacman.

# env = gym.make("MsPacman-v0")
# state = env.reset()

#You will notice that env.reset() returns a large array of numbers. To be specific, you can enter state.shape to show that our current state is represented by a 210x160x3 Tensor. This represents the height, length, and the three RGB color channels of the Atari game or, simply put, the pixels. As before, to visualize the environment you can enter:

# env.render()     #Also, as before, we can determine our possible actions by:
# env.action_space.n

#This will show that we have nine possible actions: integers 0-8. It’s important to remember that an agent should have no idea what these actions mean; its job is to learn which actions will optimize reward. But, for our sake, we can:

#env.env.get_action_meanings()

#This will show the nine possible actions the agent can chose from, represented as taking no action, and the eight possible positions of the joystick.

# Using our previous strategy, let’s see how good a random agent can perform.

# state = env.reset()
# reward, info, done = None, None, None
# while done != True:
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()

# This completely random policy will get a few hundred points, at best, and will never solve the first level.
# Continuing on, we cannot use our basic Q table algorithm because there is a total of 33,600 pixels with three RGB values that can have a range from 0 to 255. It’s easy to see that things are getting extremely complicated; this is where deep learning comes to the rescue. Using techniques such as convolutional neural networks or a DQN, a machine learning library is able to take the complex high-dimensional array of pixels, make an abstract representation, and translate that representation into a optimal action.

#In summary, you now have the basic knowledge to take Gym and start experimenting with other people's algorithms or maybe even create your own. If you would like a copy of the code used in this post to follow along with or edit, you can find the code on my GitHub.

#The field of reinforcement learning is rapidly expanding with new and better methods for solving environments—at this time, the A3C method is one of the most popular. Reinforcement learning will more than likely play an important role in the future of AI and continues to produce very interesting results.
