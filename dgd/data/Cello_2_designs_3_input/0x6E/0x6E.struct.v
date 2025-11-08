module m0x6E (input in2, in1, in3, output out);

	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in2);
	nor (\$n6_0, in1, in3);
	nor (\$n7_0, in2, \$n6_0);
	nor (\$n8_0, \$n5_0, in3);
	nor (\$n9_0, \$n8_0, \$n7_0);
	not (out, \$n9_0);

endmodule
