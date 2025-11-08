module m0x000D (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n6_0;
	wire \$n11_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;

	not (\$n8_0, in1);
	not (\$n6_0, in2);
	nor (\$n10_0, \$n6_0, \$n8_0);
	not (\$n7_0, in3);
	nor (\$n9_0, \$n7_0, in4);
	not (\$n11_0, \$n10_0);
	nor (out, \$n11_0, \$n9_0);

endmodule
