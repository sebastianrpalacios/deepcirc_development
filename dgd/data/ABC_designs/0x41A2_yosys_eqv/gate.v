module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _15 = ~_0;
assign _14 = ~_1;
assign _12 = ~_2;
assign _13 = ~_3;
assign _16 = ~_3;
assign _8 = ~(_2 | _14);
assign _5 = ~(_1 | _12);
assign _9 = ~(_15 | _16);
assign _6 = ~(_5 | _13);
assign _10 = ~(_8 | _9);
assign _7 = ~(_0 | _6);
assign _17 = ~_10;
assign _11 = ~(_7 | _17);
assign _4 = _11;
endmodule
