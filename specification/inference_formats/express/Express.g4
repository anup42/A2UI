/**
 * ANTLR4 grammar for the A2UI Express language.
 * Pinned from a2ui-project/a2ui commit
 * 2276f8cc702eaeac25ffb05be85797b2a1205c74.
 * Upstream Git blob SHA: 4f2492ae4600598d8b10e68fcd9f4292529dd653.
 */
grammar Express;

program : statement* EOF ;
statement : assignment | expression ;
assignment : (identifier | path) '=' expression ;
expression : array | map | path | check | call | variable | literal ;
array : '[' (expression (',' expression)* ','?)? ']' ;
map : '{' (map_entry (',' map_entry)* ','?)? '}' ;
map_entry : (identifier | string) ':' expression ;
path : PATH ;
check : CHECK ('(' (expression (',' expression)* ','?)? ')')? ;
call : identifier '(' (arg (',' arg)* ','?)? ')' ;
arg : named_arg | expression ;
named_arg : identifier '=' expression ;
variable : '_' | identifier ;
literal : string | NUMBER | BOOLEAN | 'null' ;
identifier : IDENTIFIER ;
string : RAW_TRIPLE_STRING | TRIPLE_STRING | RAW_STRING | STANDARD_STRING ;

RAW_TRIPLE_STRING : [rR] '"""' .*? '"""' ;
TRIPLE_STRING     : '"""' ( '\\' . | ~'\\' )*? '"""' ;
RAW_STRING        : [rR] '"' ~[\r\n"]* '"' ;
STANDARD_STRING   : '"' ( '\\' . | ~'\\' )*? '"' ;
PATH : '$' [a-zA-Z0-9_/]* ;
CHECK : '?' [a-zA-Z_] [a-zA-Z0-9_]* ;
NUMBER : '-'? [0-9]+ ('.' [0-9]+)? ;
BOOLEAN : 'true' | 'false' ;
IDENTIFIER : [a-zA-Z_] [a-zA-Z0-9_]* ;
COMMENT : ( '#' | '//' ) ~[\r\n]* -> skip ;
BLOCK_COMMENT : '/*' .*? '*/' -> skip ;
SEMICOLON : ';' -> skip ;
WS : [ \t\r\n]+ -> skip ;
