module gate(_0, _1, _2, _3, _216);
input _0, _1, _2, _3;
output _216;
wire _205, _206, _207, _208, _209, _210, _211, _212, _213;
assign _206 = ~_1;
assign _207 = ~(_0 | _2);
assign _205 = ~_3;
assign _211 = ~(_3 | _206);
assign _208 = ~(_1 | _205);
assign _212 = ~_211;
assign _209 = ~_208;
assign _213 = ~(_2 | _212);
assign _210 = ~(_207 | _209);
assign _216 = _210 | _213;
endmodule
