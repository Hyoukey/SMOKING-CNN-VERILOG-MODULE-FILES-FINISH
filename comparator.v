module comparator (
    input clk,
    input rst_n,
    input valid_in,
    input signed [11:0] data_in,  // Score from FC Layer
    output reg [2:0] decision,    // 0=Non-Smoking, 1=Smoking
    output reg valid_out
);

    // --- INTERNAL REGISTERS ---
    reg signed [11:0] buffer [0:1]; // Store Score0 and Score1
    reg buf_idx;                    // 0 or 1
    reg state;                      // 0=Reading, 1=Comparing
    reg [3:0] wait_cnt;             // Small delay for stability

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            decision <= 0;
            buffer[0] <= 0;
            buffer[1] <= 0;
            buf_idx <= 0;
            state <= 0;
            wait_cnt <= 0;
        end else begin
            
            // --- STEP 1: CAPTURE SCORES ---
            // The FC layer sends Class 0 Score first, then Class 1 Score.
            if (valid_in && state == 0) begin
                buffer[buf_idx] <= data_in;
                
                if (buf_idx == 0) begin
                    buf_idx <= 1; // Prepare for next score
                end else begin
                    buf_idx <= 0;
                    state <= 1;   // Both scores received, move to decision
                end
            end 
            
            // --- STEP 2: THE VERDICT ---
            else if (state == 1) begin
                wait_cnt <= wait_cnt + 1;
                
                // Wait 2 cycles to ensure data stability
                if (wait_cnt == 2) begin
                    // LOGIC: Is Score(Smoking) > Score(Non-Smoking)?
                    if (buffer[1] > buffer[0]) begin
                        decision <= 3'd1; // Result: Smoking
                    end else begin
                        decision <= 3'd0; // Result: Non-Smoking
                    end
                    valid_out <= 1; // Notify next module
                end
                
                // Reset after decision (Optional, prevents latching logic)
                if (wait_cnt == 6) begin
                    valid_out <= 0;
                    state <= 0;
                    wait_cnt <= 0;
                end
            end
        end
    end

endmodule