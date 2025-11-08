module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _5, _6, _7, _8, _9;
assign _12 = ~_0;
assign _10 = ~_1;
assign _11 = ~_1;
assign _13 = ~_3;
assign _5 = ~(_3 | _10);
assign _6 = ~(_2 | _11);
assign _7 = ~(_12 | _13);
assign _14 = ~_7;
assign _8 = ~(_6 | _14);
assign _9 = ~(_5 | _8);
assign _4 = _9;
endmodule
