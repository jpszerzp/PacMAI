DOCUMENTATION

Intent of Code
The main file subject to modification is the multiAgents.py, which simulate multiple agents in Pacman with different AI algorithms.
In this project I implemented three approaches: Minimax, alpha-beta pruning optimization, and Expectimax. 
I also implemented the simple Manhattan distance evaluation function and reinforcement learning agent.

Code doc
multiAgents.py -
Class that runs multiple agents of a round of Pacman game on some layout. 
The class supports adversarial algorithms and reinforcement learning, and tests their effectiveness in game.

User guide
• Navigate to correct directory (the one with layouts and test_cases folder)
• In the terminal run auto grader with command "python autograder.py"
• Observe the result of q5, as I made this an overall better evaluation function, and it is the default tested one
• To observe other agents, under same directory run pacman.py using option -p for agent and -l for layout. For example, "python pacman.py -p MinimaxAgent" plays the game with minimax agent on default layout. Name of agent is same as name of class in multiAgents.py, and name of layout is just name of corresponding file under “layouts” folder
• q1 to q5 is invented by the faculty of the course at Berkeley. For my purpose, the multiAgent.py file alone satisfies my intention to implement involved AI methods
