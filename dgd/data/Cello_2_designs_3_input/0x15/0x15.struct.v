module m0x15 (input in2, in1, in3, output out);

	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in3);
	nor (\$n6_0, in2, in1);
	nor (out, \$n6_0, \$n5_0);

endmodule
