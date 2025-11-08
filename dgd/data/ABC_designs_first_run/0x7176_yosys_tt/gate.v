module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _15 = ~_0;
assign _16 = ~_1;
assign _12 = ~_2;
assign _13 = ~_3;
assign _5 = ~(_2 | _3);
assign _6 = ~(_12 | _13);
assign _14 = ~_6;
assign _8 = ~(_6 | _15);
assign _7 = ~(_0 | _14);
assign _9 = ~(_7 | _16);
assign _17 = ~_9;
assign _10 = ~(_8 | _17);
assign _11 = ~(_5 | _10);
assign _4 = _11;
endmodule
