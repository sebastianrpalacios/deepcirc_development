module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _15 = ~_1;
assign _6 = ~(_0 | _1);
assign _12 = ~_3;
assign _16 = ~_3;
assign _8 = ~(_2 | _14);
assign _13 = ~_6;
assign _5 = ~(_2 | _12);
assign _9 = ~(_15 | _16);
assign _7 = ~(_5 | _13);
assign _10 = ~(_8 | _9);
assign _17 = ~_10;
assign _11 = ~(_7 | _17);
assign _4 = _11;
endmodule
