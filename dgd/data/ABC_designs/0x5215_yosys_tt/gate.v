module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _16, _5, _6, _7, _8, _9;
assign _16 = ~_0;
assign _13 = ~_1;
assign _15 = ~_2;
assign _14 = ~_3;
assign _9 = ~(_1 | _16);
assign _5 = ~(_0 | _13);
assign _7 = ~(_0 | _14);
assign _6 = ~(_3 | _5);
assign _10 = ~(_5 | _9);
assign _8 = ~(_7 | _15);
assign _11 = ~(_8 | _10);
assign _12 = ~(_6 | _11);
assign _4 = _12;
endmodule
