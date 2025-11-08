module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _15 = ~_1;
assign _6 = ~(_0 | _1);
assign _16 = ~_2;
assign _5 = ~(_14 | _15);
assign _10 = ~(_3 | _6);
assign _7 = ~(_5 | _6);
assign _11 = ~(_5 | _16);
assign _8 = ~(_3 | _7);
assign _17 = ~_11;
assign _9 = ~(_2 | _8);
assign _12 = ~(_10 | _17);
assign _13 = ~(_9 | _12);
assign _4 = _13;
endmodule
