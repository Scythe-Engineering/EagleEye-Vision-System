export async function populateRobotDropdown() {
    const robotFileSelect = document.getElementById("robotFileSelect");

    async function fetchAvailableRobots() {
        try {
            const response = await fetch("/get-available-robots");
            const data = await response.json();
            return data.robots;
        } catch (error) {
            console.error("Error fetching available robots:", error);
            return [];
        }
    }

    async function loadRobots() {
        const robots = await fetchAvailableRobots();
        
        // Save the currently selected value before clearing
        const previouslySelectedValue = robotFileSelect.selectedIndex > 0 ? robotFileSelect.value : null;
        
        robotFileSelect.innerHTML = "<option disabled selected>Select Robot File</option>";
        
        robots.forEach((robot) => {
            const option = document.createElement("option");
            option.value = robot;
            option.textContent = robot;
            robotFileSelect.appendChild(option);
        });

        // Restore previous selection if it still exists in the new list
        if (previouslySelectedValue && robots.includes(previouslySelectedValue)) {
            robotFileSelect.value = previouslySelectedValue;
        }
        // Otherwise, select first robot if robots are available and no previous selection
        else if (robots.length > 0 && !previouslySelectedValue) {
            robotFileSelect.selectedIndex = 1; // Index 1 because index 0 is the disabled placeholder
        }
    }

    await loadRobots();
}

    
