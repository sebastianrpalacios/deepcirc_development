module m0x8C (input in2, in1, in3, output out);

	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in3);
	nor (\$n6_0, in1, \$n5_0);
	nor (out, in2, \$n6_0);

endmodule
