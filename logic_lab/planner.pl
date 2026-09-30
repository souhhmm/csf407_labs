% optional handout extension. Run with SWI-Prolog: swipl -q -s planner.pl -g checks -t halt.
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :- connected(X,Y).
valid_move(X,Y) :- connected(X,Y).

wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

checks :-
    can_move(a,b),
    \+ can_move(a,c),
    valid_move(a,b),
    valid_move(b,c),
    \+ valid_move(a,c),
    reduce_speed,
    writeln('All six Prolog queries passed.').
