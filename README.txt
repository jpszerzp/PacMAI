DOCUMENTATION

Intent of Code
The main file subject to modification is the multiAgents.py in this case, which simulate multiple agents in Pacman with different AI algorithm.
In this project I implemented three approaches: Minimax, alpha-beta pruning optimization, and Expectimax. 
I also implemented the simple Manhattan distance evaluation function and reinforcement learning.

Code doc
multiAgents.py -
Class that runs multiple agents of a round of Pacman game on some layout. 
The class supports adversarial algorithms and reinforcement learning, and tests their effectiveness in game.

User guide
• Navigate to correct directory (the one with layouts and test_cases folder)
• In the terminal run auto grader with command “python autograder.py”
• Observe the result of q5, as this is default agent I used
• To observe other agents, open multiAgents.py, find the corresponding function (used algorithm can be told from name of the function) and use that function in evaluationFunction method of the class. Test in the same way above.
• q1 to q5 is invented by the faculty of the course at Berkeley. For my purpose, the multiAgent.py file alone satisfies my intention to implement involved AI methods.
