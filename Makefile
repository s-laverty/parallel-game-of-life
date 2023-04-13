all:
	# TODO real compile
	# Debug
	mpixlc -Wall -Werror -g -DDEBUG -c -o simulation-debug.o simulation.c
	mpixlc -Wall -Werror -g -DDEBUG -o simulation-debug simulation-debug.o
