module gate(_0, _1, _2, _3, _378);
input _0, _1, _2, _3;
output _378;
wire _368, _369, _370, _371, _372, _373, _374, _375, _376, _377;
assign _370 = ~_0;
assign _368 = ~_2;
assign _369 = ~_3;
assign _371 = ~(_369 | _370);
assign _372 = ~_371;
assign _374 = ~(_2 | _371);
assign _373 = ~(_368 | _372);
assign _375 = ~(_1 | _374);
assign _376 = ~_375;
assign _377 = ~(_373 | _376);
assign _378 = _377;
endmodule
