module m0x0026 (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n11_0;
	wire \$n14_0;
	wire \$n9_0;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n13_0;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n6_1;

	not (\$n7_0, in2);
	nor (\$n10_0, \$n7_0, \$n6_0);
	not (\$n6_0, in4);
	not (\$n6_1, in4);
	not (\$n8_0, in3);
	nor (\$n12_0, \$n8_0, \$n6_1);
	not (\$n9_0, in1);
	nor (\$n13_0, \$n9_0, \$n12_0);
	not (\$n14_0, \$n13_0);
	nor (\$n11_0, \$n10_0, in3);
	nor (out, \$n11_0, \$n14_0);

endmodule
