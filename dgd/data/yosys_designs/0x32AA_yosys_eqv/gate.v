module gate(_0, _1, _2, _3, _395);
input _0, _1, _2, _3;
output _395;
wire _390, _391, _392, _393, _394;
assign _391 = ~(_0 | _1);
assign _393 = ~(_0 | _2);
assign _390 = ~_3;
assign _392 = ~(_390 | _391);
assign _394 = ~(_392 | _393);
assign _395 = _394;
endmodule
