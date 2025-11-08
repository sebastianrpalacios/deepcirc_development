module m0x7F (input in2, in1, in3, output out);

	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	nor (\$n5_0, in2, in3);
	not (\$n6_0, \$n5_0);
	nor (\$n7_0, in1, \$n6_0);
	not (out, \$n7_0);

endmodule
