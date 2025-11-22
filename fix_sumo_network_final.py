"""Final fix for SUMO network - remove uncontrolled and generate proper signal plans"""
import re
import subprocess
import os
import xml.etree.ElementTree as ET

net_file = "sumo_scenarios/intersection/intersection.net.xml"
add_file = "sumo_scenarios/intersection/intersection.add.xml"
config_file = "sumo_scenarios/intersection/intersection.sumocfg"

print("Step 1: Removing uncontrolled attributes...")
with open(net_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all uncontrolled="1" attributes
content = re.sub(r' uncontrolled="1"', '', content)

with open(net_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Removed uncontrolled attributes")

print("\nStep 2: Processing network with netconvert to generate signal plans...")
result = subprocess.run(
    ["netconvert", "--net-file", net_file,
     "--tls.guess-signals", "true",
     "--tls.default-type", "static",
     "--output-file", net_file],
    capture_output=True,
    text=True,
    timeout=30
)

if result.returncode == 0:
    print("✅ Network processed successfully")
else:
    print(f"⚠️ netconvert warning: {result.stderr[:200]}")

print("\nStep 3: Checking if signal plans are embedded in network...")
tree = ET.parse(net_file)
root = tree.getroot()
tls_in_net = root.findall(".//tlLogic")

if tls_in_net:
    print(f"✅ Found {len(tls_in_net)} signal plans in network file")
    # Extract to additional file
    additional = ET.Element("additional")
    for tl in tls_in_net:
        # Create a copy
        new_tl = ET.Element("tlLogic")
        new_tl.set("id", tl.get("id"))
        new_tl.set("type", tl.get("type", "static"))
        new_tl.set("programID", tl.get("programID", "0"))
        new_tl.set("offset", tl.get("offset", "0"))
        
        for phase in tl.findall("phase"):
            new_phase = ET.SubElement(new_tl, "phase")
            new_phase.set("duration", phase.get("duration"))
            new_phase.set("state", phase.get("state"))
        
        additional.append(new_tl)
        # Remove from network (signal plans should be in additional file)
        root.remove(tl)
    
    # Save network without embedded signal plans
    tree.write(net_file, encoding="utf-8", xml_declaration=True)
    
    # Save additional file
    tree_add = ET.ElementTree(additional)
    ET.indent(tree_add, space="  ")
    tree_add.write(add_file, encoding="utf-8", xml_declaration=True)
    print(f"✅ Extracted signal plans to {add_file}")
else:
    print("⚠️ No signal plans found in network file")
    print("Creating signal plans manually based on network structure...")
    
    # Get junctions with traffic lights
    junctions = root.findall("junction[@type='traffic_light']")
    print(f"Found {len(junctions)} traffic light junctions")
    
    additional = ET.Element("additional")
    
    for junction in junctions:
        tl_id = junction.get("id")
        inc_lanes = junction.get("incLanes", "").split()
        num_links = len(inc_lanes)
        
        print(f"  {tl_id}: {num_links} incoming lanes")
        
        tl_logic = ET.SubElement(additional, "tlLogic")
        tl_logic.set("id", tl_id)
        tl_logic.set("type", "static")
        tl_logic.set("programID", "0")
        tl_logic.set("offset", "0")
        
        # Create signal states matching number of links
        # For 4 links, use 4-state plan
        if num_links == 4:
            phases = [
                ("GGrr", 30),
                ("yyrr", 3),
                ("rrGG", 30),
                ("rryy", 3),
            ]
        else:
            # Generic: create states for all links
            half = num_links // 2
            green1 = "G" * half + "r" * (num_links - half)
            yellow1 = "y" * half + "r" * (num_links - half)
            green2 = "r" * (num_links - half) + "G" * half
            yellow2 = "r" * (num_links - half) + "y" * half
            
            phases = [
                (green1, 30),
                (yellow1, 3),
                (green2, 30),
                (yellow2, 3),
            ]
        
        for state, duration in phases:
            phase = ET.SubElement(tl_logic, "phase")
            phase.set("duration", str(duration))
            phase.set("state", state)
    
    tree_add = ET.ElementTree(additional)
    ET.indent(tree_add, space="  ")
    tree_add.write(add_file, encoding="utf-8", xml_declaration=True)
    print(f"✅ Created signal plans in {add_file}")

print("\nStep 4: Updating config file to include additional file...")
tree_config = ET.parse(config_file)
root_config = tree_config.getroot()

# Ensure additional file is included
input_elem = root_config.find("input")
if input_elem is None:
    input_elem = ET.SubElement(root_config, "input")

# Check if additional-files element exists
add_files_elem = input_elem.find("additional-files")
if add_files_elem is None:
    add_files_elem = ET.SubElement(input_elem, "additional-files")
    add_files_elem.set("value", "intersection.add.xml")
else:
    add_files_elem.set("value", "intersection.add.xml")

tree_config.write(config_file, encoding="utf-8", xml_declaration=True)
print("✅ Updated config file")

print("\n✅ Network fix complete!")
print("\nNext steps:")
print("1. Refresh your Streamlit app")
print("2. Click 'Start SUMO Simulation'")
print("3. Traffic lights should now be detected!")

