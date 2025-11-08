module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _15, _16, _17, _18, _19;
assign _18 = ~_1;
assign _15 = ~(_0 | _2);
assign _16 = ~(_2 | _15);
assign _17 = ~(_0 | _15);
assign _19 = ~(_17 | _18);
assign _13 = _16 | _19;
endmodule
