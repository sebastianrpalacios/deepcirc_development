module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _11, _12, _13, _14, _5, _6, _7, _8;
assign _11 = ~_0;
assign _12 = ~_1;
assign _7 = ~(_1 | _2);
assign _13 = ~_3;
assign _5 = ~(_2 | _11);
assign _8 = ~(_7 | _13);
assign _6 = ~(_5 | _12);
assign _14 = ~_8;
assign _4 = _6 | _14;
endmodule
