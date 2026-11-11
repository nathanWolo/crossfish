# Crossfish explainer: narration script

Generated from `src/scenes/*.js` by `tools/export_script.mjs`. Edit the scenes, not this file. 2013 words.

## Cold open

This is a game of tic-tac-toe, played on a board made of nine smaller tic-tac-toe boards.

Both sides here are Crossfish, a program I wrote to play it. It gets a tenth of a second per move. In that time it examines around seven hundred thousand positions, and reads the game fifteen to twenty moves ahead.

It started as a two-week Python project for a university tournament. Two years later, it reached number one on a worldwide ladder.

This is how it works. And it is also the story of a hash table that never worked, a missing compiler flag, and a neural network small enough to fit in a text box.


## The rules

The rules take thirty seconds. The first move can go anywhere. After that, the cell you pick decides which board your opponent has to play in.

X plays in the top right cell, so O must play in the top right board. O picks the bottom left cell, and X is sent to the bottom left board.

Get three in a row on a small board and you win that board.

If you are sent to a board that is already finished, you can play anywhere.

Win three small boards in a row, and you win the game.

That sending rule is what makes this hard. Every move is two decisions at once: where do I play, and where do I let my opponent play? A move that looks great locally can hand them exactly the board they wanted.

People find this game confusing. Computers are well suited to it, for the same reason they are suited to chess: you can search.


## Chapter 1: Search

January 2024. The UVic AI club is running an Ultimate Tic-Tac-Toe bot tournament, and I have about two weeks to write an entry.

The core idea is older than computers. From the current position, list every move you could make. For each of those, list every reply. Keep going, and you get a tree.

At the bottom, give each position a score: high if it is good for X, low if it is good for O. Then work back up. On O's turn, take the lowest score below. On X's turn, take the highest. This is minimax. It finds the best move, assuming both sides play as well as they can.

The problem is size. A typical position has about nine legal moves, so every extra move of lookahead multiplies the work by nine.

The fix is called alpha-beta pruning. Say X has already found a move worth three. While checking the next move, O finds a reply worth only one. We can stop right there. X would never choose this branch, so whatever else is under it does not matter.

How much you can skip depends on the order you look at moves. Try the best move first, and almost everything else gets cut. So a large part of engine design is guessing well. Try the move that was best the last time you saw this position. Try moves that win a small board. Avoid moves that hand over a free move.

In that first week, one change, trying the remembered best move first, was worth more than four hundred Elo by itself.

And since the clock is fixed, the engine searches one move deep, then two, then three, each pass using the previous one to order its moves, until time runs out.

Two weeks in, that Python bot won the tournament.


## Chapter 2: Speed

The next stop was CodinGame, which runs a permanent online ladder for this game. The limit there is one hundred milliseconds per move, so I rewrote the engine in C++.

Speed starts with how you store the board. Each small board is just two nine-bit numbers: one bit per cell for X, and one bit per cell for O.

And a small board only has three to the power of nine possible states. That is nineteen thousand, six hundred and eighty-three. Few enough to work out everything about every one of them in advance: whether it is won, which cells would complete a line, what it is worth. While searching, the engine computes almost nothing about a small board. It looks it up.

In March 2024 that engine reached Legend, the top league on the site. And then I left it alone for more than two years.


## Chapter 3: The autopsy

August 2026. I came back to the project, this time working with AI coding agents, and the first job was an autopsy of the old code.

To explain what it found, I need one more piece of search. Different move orders often reach exactly the same position. So engines keep a transposition table: a big hash table that remembers positions they have already searched, and what the answer was.

The hash is built by giving every possible mark on the board its own random sixty-four bit key, and combining the keys of the marks that are present. Make a move, mix in one more key. It costs almost nothing.

In my engine, those keys were filled in by the constructor. Into local variables, which happened to have the same names as the real ones. The real keys were never set. So every position in the game had the same hash: zero.

Every position read and wrote the same single slot. Back in 2024 I had even measured that switching the table's cutoffs off made the engine stronger, and concluded they were a bad idea. They were bad because the table was full of garbage. The engine had reached the top league with its memory completely broken.

Fixing it was worth about a hundred and twenty Elo. One of the biggest gains in the whole project was a bug.


## Chapter 4: How do you know?

That raised an uncomfortable question. If a bug that large could hide for two years, how would I know whether any change I make is an improvement?

You cannot tell by watching games. A genuine five Elo improvement moves the expected score from fifty percent to fifty point seven. No one can see that.

So every change goes through the same loop. Freeze the current engine. Change one thing in a copy. Then let them play thousands of games from balanced openings, each opening twice with the colours swapped. A statistical test called S P R T watches the results, and stops as soon as it is confident either way.

It is humbling. One time, a copy of the engine with no changes at all was tested against itself. After three thousand games it looked seven Elo stronger. That was pure noise.

And most ideas fail. Null-move pruning, razoring, a bigger hash table: all standard in chess engines, and all of them made this one worse. A line from the project log: chess folklore is not a patch list.

The ones that passed were kept. Through September the engine got faster and smarter in small, measured steps.


## Chapter 5: Evaluation

So far I have skipped the hardest part. At the bottom of the tree, the search needs a number: how good is this position? The function that answers is called the evaluation.

The first version was written by hand. Count the small boards each side has won. Add a bonus for the centre board, for two in a row, and a few more. About ten weights, tuned on games the engine played against itself.

The obvious upgrade is a neural network, and the first attempts taught me the central trade-off of this whole field. One larger network was clearly a better judge of positions: forty-eight Elo stronger when both engines searched to the same depth. On the clock, in real games, it lost by nearly three hundred. It was twelve times slower.

Every microsecond spent judging one position is a microsecond not spent searching others. A better evaluation that you cannot afford is a worse engine.

So the first network that shipped was tiny. It only nudged the handwritten score, and it gained seven Elo. Progress. But a much bigger gain was hiding somewhere else entirely.


## Chapter 6: The missing flag

All of my testing ran on my own machines, compiled with full optimization. CodinGame compiles your code itself. One day, we checked how.

No optimization flag at all. The bot I had been carefully tuning at seven hundred thousand positions per move was running on the ladder at about a hundred and fifty thousand.

Every result I had ever seen there came from an engine running at a fifth of its speed. The fix was a few lines that force optimization from inside the source file. It was worth a hundred and sixty Elo. And no test on my machine could have found it, because on my machine nothing was wrong.


## Chapter 7: The network

By late September, the handwritten evaluation and its helper networks had been pushed about as far as they would go. The replacement was built around one fact from earlier: a small board only has nineteen thousand patterns.

Here is the design. Take one small board, sitting in one of the nine places. A small network, the encoder, reads its nine cells and outputs sixty-four numbers that describe that pattern in that place.

Now the trick. There are only nineteen thousand patterns and nine places. So when the program starts, it runs the encoder on every pattern in every place, and stores the answers in a table. That takes a fraction of a second, once. After that the encoder never runs again.

During search, the engine keeps a running total: the sum of the nine table rows for the nine small boards. When a move is made, only one small board changes. So subtract its old row, add its new row, and the most expensive layer of the network is already up to date.

What is left is a tiny network on top, which turns that total into a score. Chess engines call this idea N N U E. This version is shaped around this particular game: it sees each small board as a whole pattern, never as nine separate cells.

The current network has fifty-one thousand parameters. It was trained on about eighty million positions from the engine's own games, each one labelled with the verdict of a deeper search.

The first version made the engine slower. It lost about half its speed, and looked one move less deep. It was also two hundred and seventy Elo stronger. That is the largest single step in the project.

After that submission, Crossfish finished its placement games at number one on the ladder.


## Chapter 8: Fitting in the box

There was one more constraint. CodinGame accepts a single source file of at most one hundred thousand characters. The engine, and every weight of the network, has to fit inside that.

But the limit counts characters, not bytes. So the weights are compressed, and then written out fifteen bits at a time, mostly as Chinese characters. The neural network is a string of forty-two thousand of them, sitting in the middle of the source code.

This is the other reason for the pattern trick. The tables the engine uses while searching take twenty-five megabytes. The source file only carries the small network that generates them.

And the final submission is not even C++. It is a Python script that unpacks a binary built ahead of time with a different compiler, and runs it. That was worth another seven percent of speed.


## Chapter 9: Diminishing returns

After the big jump, the returns shrank. Retraining the network on its own games: plus fifty. More data: plus sixteen. Cleaner labels: plus nine. A wider encoder: plus fourteen. Each of those needed thousands of games just to detect.

The ladder moved as well. Other authors kept improving their bots, and the latest version of Crossfish placed between sixth and ninth, not first.

Looking back, the biggest gains did not come from the most sophisticated ideas. A hash table whose keys were never filled in. A compiler flag that was never set. And a testing loop strict enough to tell a real improvement from noise.

The code is linked below, along with the log of every experiment, including the ones that failed. Thanks for watching.

