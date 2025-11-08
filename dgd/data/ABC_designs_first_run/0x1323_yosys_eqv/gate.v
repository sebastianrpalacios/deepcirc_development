module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _5, _6, _7, _8, _9;
assign _10 = ~_0;
assign _13 = ~_2;
assign _11 = ~_3;
assign _5 = ~(_3 | _10);
assign _6 = ~(_0 | _11);
assign _7 = ~(_1 | _5);
assign _12 = ~_7;
assign _8 = ~(_6 | _12);
assign _9 = ~(_8 | _13);
assign _4 = _9;
endmodule
