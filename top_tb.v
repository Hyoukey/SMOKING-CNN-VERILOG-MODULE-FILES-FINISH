`timescale 1ns/1ps

module top_tb();

reg clk, rst_n;
wire [6:0] hex;
wire finish;

// Module Instantiation
top u_top(
  .clk(clk),
  .rst_n(rst_n),
  .hex(hex),
  .finish(finish)
);

// Clock generation (10ns period = 100MHz)
always #5 clk = ~clk;

// Test Stimulus
initial begin
  // Initialize signals
  clk = 0;
  rst_n = 1;

  // Reset Sequence
  #10 rst_n = 0; // Apply Reset
  #10 rst_n = 1; // Release Reset
  
  // The simulation will now run automatically because the image 
  // is hardcoded inside conv1_buf.
  // We just wait.  <-- FIXED: Commented out this line
  
  // (Optional) Stop after a long time if it doesn't stop itself
  #2000000; 
  
  $display("Simulation timed out!");
  $stop;
end

endmodule