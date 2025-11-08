module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _13 = ~_1;
assign _15 = ~_3;
assign _7 = ~(_0 | _3);
assign _5 = ~(_3 | _13);
assign _8 = ~(_14 | _15);
assign _9 = ~(_1 | _7);
assign _6 = ~(_2 | _5);
assign _16 = ~_9;
assign _10 = ~(_8 | _16);
assign _4 = _6 | _10;
endmodule
