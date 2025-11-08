module gate(_0, _1, _2, _3, _460);
input _0, _1, _2, _3;
output _460;
wire _453, _454, _455, _456, _457, _458, _459;
assign _454 = ~_0;
assign _453 = ~_1;
assign _455 = ~(_3 | _453);
assign _456 = ~_455;
assign _458 = ~(_2 | _455);
assign _457 = ~(_454 | _456);
assign _459 = ~(_457 | _458);
assign _460 = _459;
endmodule
