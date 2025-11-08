module m0x8C (input in2, in1, in3, output out);

	wire \$new_n6__0;
	wire \$new_n5__0;

	not (\$new_n5__0, in3);
	nor (\$new_n6__0, \$new_n5__0, in1);
	nor (out, \$new_n6__0, in2);

endmodule
