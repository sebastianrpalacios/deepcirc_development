module gate(_0, _1, _2, _3, _315);
input _0, _1, _2, _3;
output _315;
wire _307, _308, _309, _310, _311, _312, _313, _314;
assign _309 = ~_0;
assign _308 = ~_1;
assign _307 = ~_3;
assign _311 = ~(_3 | _308);
assign _310 = ~(_1 | _307);
assign _312 = ~(_309 | _311);
assign _313 = ~(_2 | _312);
assign _314 = ~(_310 | _313);
assign _315 = _314;
endmodule
