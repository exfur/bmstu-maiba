for ($i = 1; $i -le 15; $i++) {
    Write-Host "`n====================================" -ForegroundColor Cyan
    Write-Host "Running iteration $i of 15" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan
    uv run generate_multimodal_data.py -v $i --no-ai
}
Read-Host -Prompt "Press Enter to exit"