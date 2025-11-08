module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _17 = ~_1;
assign _16 = ~_2;
assign _13 = ~_3;
assign _15 = ~_3;
assign _8 = ~(_3 | _16);
assign _5 = ~(_1 | _13);
assign _7 = ~(_2 | _15);
assign _6 = ~(_5 | _14);
assign _9 = ~(_7 | _17);
assign _18 = ~_9;
assign _10 = ~(_8 | _18);
assign _4 = _6 | _10;
endmodule
