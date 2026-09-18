import win32com.client
pl = win32com.client.Dispatch("PrairieLink.Application")
pl.Connect()

print("Sending command ...")
success = pl.SendScriptCommands('-TSeries')  # Returns when command is sent, not when command finishes
print(f"Command sent: {success}")
