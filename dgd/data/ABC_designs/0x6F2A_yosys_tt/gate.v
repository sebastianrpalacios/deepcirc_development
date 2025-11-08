module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _12, _13, _14, _5, _6, _7, _8, _9;
assign _12 = ~_2;
assign _5 = ~(_1 | _2);
assign _13 = ~_3;
assign _7 = ~(_1 | _12);
assign _6 = ~(_3 | _5);
assign _8 = ~(_0 | _13);
assign _14 = ~_8;
assign _9 = ~(_7 | _14);
assign _4 = _6 | _9;
endmodule
