<#
.SYNOPSIS
A PowerShell script to interact with the Telegram Bot API.

.DESCRIPTION
This script provides basic functions to get bot information, get updates, and send messages using a Telegram bot.

.PARAMETER BotToken
The token for your Telegram bot. You can get this from the BotFather.

.PARAMETER ChatId
The ID of the chat you want to send messages to.

.EXAMPLE
PS > .\telegram_bot.ps1 -BotToken "YOUR_BOT_TOKEN" -ChatId "YOUR_CHAT_ID"
This example runs the script with the specified bot token and chat ID.

.NOTES
To get a BotToken:
1. Open Telegram and search for "BotFather".
2. Start a chat with BotFather and send the /newbot command.
3. Follow the instructions to create a new bot.
4. BotFather will give you a token.

To get a ChatId:
1. After you have your bot token, run this script with only the -BotToken parameter.
2. The script will call the Get-BotUpdates function.
3. Send a message to your bot from the chat you want to get the ID for.
4. Look at the output of the Get-BotUpdates function. You will see a "chat" object with an "id" field. This is your ChatId.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$BotToken,

    [Parameter(Mandatory=$false)]
    [string]$ChatId
)

function Get-BotInfo {
    <#
    .SYNOPSIS
    Gets information about the bot.
    .DESCRIPTION
    Calls the getMe method of the Telegram Bot API to get basic information about the bot.
    .EXAMPLE
    PS > Get-BotInfo
    #>
    $uri = "https://api.telegram.org/bot$BotToken/getMe"
    try {
        return Invoke-RestMethod -Uri $uri -Method Get
    }
    catch {
        Write-Error "Failed to get bot info. Please check your BotToken and internet connection."
    }
}

function Get-BotUpdates {
    <#
    .SYNOPSIS
    Gets updates for the bot.
    .DESCRIPTION
    Calls the getUpdates method of the Telegram Bot API to get recent updates.
    .EXAMPLE
    PS > Get-BotUpdates
    #>
    $uri = "https://api.telegram.org/bot$BotToken/getUpdates"
    try {
        return Invoke-RestMethod -Uri $uri -Method Get
    }
    catch {
        Write-Error "Failed to get updates. Please check your BotToken and internet connection."
    }
}

function Send-BotMessage {
    <#
    .SYNOPSIS
    Sends a message to a chat.
    .DESCRIPTION
    Calls the sendMessage method of the Telegram Bot API to send a message to a specified chat.
    .PARAMETER Message
    The message to send.
    .EXAMPLE
    PS > Send-BotMessage -Message "Hello, world!"
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message
    )

    $uri = "https://api.telegram.org/bot$BotToken/sendMessage"
    $body = @{
        chat_id = $ChatId
        text = $Message
    }

    try {
        return Invoke-RestMethod -Uri $uri -Method Post -Body $body
    }
    catch {
        Write-Error "Failed to send message. Please check your BotToken, ChatId, and internet connection."
    }
}

# --- Example Usage ---

# 1. Get bot information to verify the token
Write-Host "Getting bot information..."
$botInfo = Get-BotInfo
if ($botInfo) {
    Write-Host "Bot Name: $($botInfo.result.first_name)"
    Write-Host "Bot Username: $($botInfo.result.username)"
}
else {
    # Exit if we can't get bot info
    return
}

# 2. Check if a ChatId was provided
if ($ChatId) {
    # Send a test message
    Write-Host "Sending a test message to ChatId: $ChatId"
    Send-BotMessage -Message "Hello from the PowerShell Telegram Bot!"
}
else {
    # Get updates to help the user find their ChatId
    Write-Host "No ChatId provided. Getting updates to help you find it."
    Write-Host "Please send a message to your bot in Telegram."
    # Pause for a moment to allow the user to send a message
    Start-Sleep -Seconds 10
    $updates = Get-BotUpdates
    if ($updates.result.Count -gt 0) {
        Write-Host "Recent updates:"
        $updates.result | ForEach-Object {
            Write-Host "ChatId: $($_.message.chat.id), User: $($_.message.from.first_name), Message: $($_.message.text)"
        }
        Write-Host "Please re-run the script with the -ChatId parameter."
    }
    else {
        Write-Host "No updates found. Please make sure you have sent a message to your bot."
    }
}
