module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _13 = ~_2;
assign _15 = ~_3;
assign _6 = ~(_2 | _14);
assign _5 = ~(_0 | _13);
assign _7 = ~(_1 | _6);
assign _8 = ~(_5 | _7);
assign _16 = ~_8;
assign _9 = ~(_8 | _15);
assign _10 = ~(_3 | _16);
assign _4 = _9 | _10;
endmodule
