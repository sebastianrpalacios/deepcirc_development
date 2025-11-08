module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _5, _6, _7, _8, _9;
assign _13 = ~_1;
assign _14 = ~_2;
assign _16 = ~_2;
assign _17 = ~_3;
assign _5 = ~(_0 | _13);
assign _6 = ~(_1 | _14);
assign _9 = ~(_0 | _16);
assign _7 = ~(_3 | _5);
assign _18 = ~_9;
assign _15 = ~_7;
assign _10 = ~(_17 | _18);
assign _8 = ~(_6 | _15);
assign _4 = _8 | _10;
endmodule
