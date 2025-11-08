module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _15, _16, _17, _18, _19;
assign _19 = ~_0;
assign _16 = ~(_0 | _1);
assign _18 = ~_1;
assign _17 = ~(_2 | _16);
assign _15 = ~(_18 | _19);
assign _13 = _15 | _17;
endmodule
