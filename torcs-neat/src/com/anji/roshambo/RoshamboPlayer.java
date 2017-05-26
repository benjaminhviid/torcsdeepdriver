package com.anji.roshambo;

import com.anji.tournament.IteratedPlayer;

/**
 * http://www.cs.unimaas.nl/~donkers/games/roshambo03/
 */
public interface RoshamboPlayer extends IteratedPlayer {

/**
 * enumerated value for rock
 */
int ROCK = 0;

/**
 * enumerated value for paper
 */
int PAPER = 1;

/**
 * enumerated value for scissors
 */
int SCISSORS = 2;

/**
 * enumerated value for draw (aka, tie)
 */
int DRAW = 0;

/**
 * enumerated value for win
 */
int WIN = 1;

/**
 * enumerated value for loss
 */
int LOSS = -1;

/**
 * Initialize a new match against an unknown opponent. The length of the match is specified in
 * the paramater "trials". This function is always called before any match is played.
 * @param trials
 */
void reset(int trials);

/**
 * Store the opponent's choice and the outcome of the latest move. This function is called after
 * every move. move is one of ROCK, PAPER, SCISSORS. score is one of DRAW, WIN, LOSS.
 * @param move
 * @param score
 */
void storeMove(int move, int score);

/**
 * Produce your next move.
 * @return one of ROCK, PAPER, SISSORS
 */
int nextMove();

/**
 * @return the name (and version) of this subject.
 */
String getPlayerId();

/**
 * @return the author's name of this subject.
 */
String getAuthor();
}
