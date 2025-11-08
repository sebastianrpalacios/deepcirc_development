module m0x1D (input in2, in1, in3, output out);

	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in2);
	nor (\$n6_0, in2, in1);
	nor (\$n7_0, \$n5_0, in3);
	nor (out, \$n7_0, \$n6_0);

endmodule
