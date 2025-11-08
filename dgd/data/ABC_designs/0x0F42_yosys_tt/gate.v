module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _19, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _16 = ~_0;
assign _15 = ~_1;
assign _13 = ~_2;
assign _17 = ~_3;
assign _8 = ~(_1 | _16);
assign _5 = ~(_3 | _13);
assign _9 = ~(_2 | _17);
assign _18 = ~_8;
assign _6 = ~(_5 | _14);
assign _19 = ~_9;
assign _7 = ~(_6 | _15);
assign _10 = ~(_18 | _19);
assign _4 = _7 | _10;
endmodule
