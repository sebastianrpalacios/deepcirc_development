module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _44, _45, _46, _47, _48, _49, _50;
assign _44 = ~_0;
assign _45 = ~(_1 | _2);
assign _47 = ~(_1 | _3);
assign _46 = ~(_2 | _45);
assign _48 = ~(_3 | _45);
assign _50 = ~(_46 | _47);
assign _49 = ~(_44 | _48);
assign _4 = _49 | _50;
endmodule
