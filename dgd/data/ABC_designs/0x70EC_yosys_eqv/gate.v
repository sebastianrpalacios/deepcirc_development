module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _12, _13, _14, _15, _5, _6, _7, _8, _9;
assign _12 = ~_0;
assign _13 = ~_0;
assign _14 = ~_3;
assign _6 = ~(_2 | _3);
assign _5 = ~(_2 | _12);
assign _7 = ~(_13 | _14);
assign _8 = ~(_1 | _6);
assign _15 = ~_8;
assign _9 = ~(_7 | _15);
assign _4 = _5 | _9;
endmodule
