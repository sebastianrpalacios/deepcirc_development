module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _14, _15, _16, _17, _18, _19;
assign _19 = ~_1;
assign _15 = ~(_1 | _2);
assign _18 = ~_2;
assign _16 = ~(_18 | _19);
assign _17 = ~(_0 | _16);
assign _14 = ~(_15 | _17);
assign _13 = _14;
endmodule
