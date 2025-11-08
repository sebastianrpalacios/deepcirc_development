module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _14 = ~_1;
assign _8 = ~(_0 | _1);
assign _15 = ~_2;
assign _13 = ~_3;
assign _16 = ~_8;
assign _9 = ~(_3 | _15);
assign _5 = ~(_2 | _13);
assign _17 = ~_9;
assign _6 = ~(_0 | _5);
assign _10 = ~(_16 | _17);
assign _7 = ~(_6 | _14);
assign _4 = _7 | _10;
endmodule
